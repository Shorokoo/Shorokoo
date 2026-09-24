using Python.Runtime;
using Shorokoo.Core.Backends;
using Shorokoo.PythonHost;

namespace Shorokoo.PyTorch;

/// <summary>
/// A value of the PyTorch backend: a torch tensor, a string tensor (a numpy array of Python
/// strings, since torch has none), or a sequence (a Python list of either).
///
/// <para>What a value is — its kind, element type, shape, and for a host tensor the address and
/// size of its buffer — is read once, when it is made, and kept: none of it can change, and
/// reading it again would take the interpreter lock for every question.</para>
///
/// <para><b>The span accessors read the tensor's own memory.</b> A host tensor this wraps is always
/// contiguous and owns its storage (the backend makes it so before wrapping it), and it lives for
/// as long as this value holds its reference — so a span stays valid until the value is released,
/// exactly as an ONNX Runtime value's does, and a caller keeps the value alive across its use of
/// the span the same way. No lock is needed to read it: the memory is torch's, not the
/// interpreter's.</para>
/// </summary>
public sealed class TorchTensorValue : IShorokooTensorValue
{
    private readonly PyObject _value;
    private readonly long[] _shape;
    private readonly IntPtr _address;
    private readonly long _byteCount;
    private readonly ShorokooTensorElementType _elementType;
    private readonly bool _isHost;
    private int _released;

    private TorchTensorValue(
        PyObject value, ShorokooOnnxValueType valueType, ShorokooTensorElementType elementType,
        long[] shape, bool isHost, IntPtr address, long byteCount)
    {
        _value = value;
        ValueType = valueType;
        _elementType = elementType;
        _shape = shape;
        _isHost = isHost;
        _address = address;
        _byteCount = byteCount;
    }

    /// <summary>
    /// Wraps <paramref name="value"/>, taking over the reference, and reads what it is. Called
    /// holding the interpreter lock. <paramref name="emptySequenceElementType"/> is what an empty
    /// sequence reports as its element type, having no element to ask.
    /// </summary>
    internal static TorchTensorValue Wrap(
        TorchRuntime runtime, PyObject value, ShorokooTensorElementType emptySequenceElementType)
    {
        using var description = runtime.Describe.Invoke(value);
        return Wrap(value, description, emptySequenceElementType);
    }

    /// <summary>Wraps <paramref name="value"/> by a description <c>runtime.describe</c> already
    /// made of it. Called holding the interpreter lock.</summary>
    internal static TorchTensorValue Wrap(
        PyObject value, PyObject description, ShorokooTensorElementType emptySequenceElementType)
    {
        var kind = description[0].As<int>();
        var code = description[1].As<int>();
        using var dims = description[2];
        var shape = new long[(int)dims.Length()];
        for (int i = 0; i < shape.Length; i++) shape[i] = dims[i].As<long>();
        var isHost = description[3].As<bool>();
        var address = new IntPtr(description[4].As<long>());
        var byteCount = description[5].As<long>();
        return kind switch
        {
            0 => new TorchTensorValue(value, ShorokooOnnxValueType.Tensor, (ShorokooTensorElementType)code,
                shape, isHost, address, byteCount),
            1 => new TorchTensorValue(value, ShorokooOnnxValueType.Sequence,
                code == 0 ? emptySequenceElementType : (ShorokooTensorElementType)code, shape, true, IntPtr.Zero, 0),
            _ => throw new NotSupportedException(
                "The PyTorch backend produced an absent optional value, which has no representation here yet."),
        };
    }

    /// <summary>The Python object this wraps, refused once released: a released value's reference
    /// is gone, and the object with it.</summary>
    internal PyObject Value => Volatile.Read(ref _released) == 0 ? _value : throw Released();

    private static ObjectDisposedException Released() => new(
        nameof(TorchTensorValue),
        "This PyTorch value has been released -- the tensor it belonged to was deleted, or consumed by "
        + "a run whose backend released it, or it was handed into a sequence that owns it now, or it "
        + "was disposed -- so nothing may read it through this handle.");

    public ShorokooOnnxValueType ValueType { get; }

    public ShorokooTensorElementType ElementType => _elementType;

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
                "This tensor is in a CUDA device's memory, not host memory, so it cannot be read "
                + "directly. CopyTensorToHost brings it home.");
        if (ValueType != ShorokooOnnxValueType.Tensor)
            throw new InvalidOperationException(
                $"This value holds a {ValueType}, not a tensor, so it has no element buffer to read. "
                + "Read a sequence through its elements instead.");
        if (_elementType == ShorokooTensorElementType.String)
            throw new InvalidOperationException(
                "A string tensor has no element buffer; read it with GetStringTensorData.");
        if (_byteCount % System.Runtime.CompilerServices.Unsafe.SizeOf<T>() != 0)
            throw new InvalidOperationException(
                $"A {_elementType} tensor of {_byteCount} bytes cannot be read as {typeof(T).Name}.");
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

    /// <summary>The host address and size of a host tensor's buffer, for the backend's own copies.</summary>
    internal (IntPtr Address, long ByteCount) HostBuffer
    {
        get
        {
            if (Volatile.Read(ref _released) != 0) throw Released();
            return (_address, _byteCount);
        }
    }

    public IReadOnlyList<string> GetStringTensorData()
    {
        if (_elementType != ShorokooTensorElementType.String)
            throw new InvalidOperationException($"This is a {_elementType} tensor, not a string tensor.");
        var runtime = RuntimeOf();
        using (PythonRuntime.Gil())
        {
            using var list = runtime.StringList.Invoke(Value);
            var strings = new string[(int)list.Length()];
            for (int i = 0; i < strings.Length; i++) strings[i] = list[i].As<string>();
            return strings;
        }
    }

    public int GetValueCount()
    {
        if (ValueType != ShorokooOnnxValueType.Sequence)
            throw new InvalidOperationException($"This value holds a {ValueType}, not a sequence.");
        return checked((int)_shape[0]);
    }

    /// <summary>The element at <paramref name="index"/>, as a value of its own: a copy, so writing to
    /// it cannot reach into this sequence.</summary>
    public IShorokooTensorValue GetValue(int index)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(index, GetValueCount());
        var runtime = RuntimeOf();
        using (PythonRuntime.Gil())
        {
            var element = runtime.SequenceElement.Invoke(Value, new PyInt(index));
            return Wrap(runtime, element, _elementType);
        }
    }

    public ShorokooTensorElementType GetSequenceElementType()
    {
        if (ValueType != ShorokooOnnxValueType.Sequence)
            throw new InvalidOperationException($"This value holds a {ValueType}, not a sequence.");
        return _elementType;
    }

    /// <summary>Drops this value's reference, once: a second call does nothing, and every read
    /// afterwards is refused.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _released, 1) != 0) return;
        PythonRuntime.Release(_value);
    }

    /// <summary>Marks this released and drops its reference, for a value handed into a sequence:
    /// the sequence holds a reference of its own, so the tensor lives on there.</summary>
    internal void HandedOver() => Dispose();

    private static TorchRuntime RuntimeOf()
        => TorchRuntime.Current
            ?? throw new InvalidOperationException("PyTorch has not been started in this process.");
}

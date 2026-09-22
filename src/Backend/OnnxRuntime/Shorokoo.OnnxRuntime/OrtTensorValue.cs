using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Backends;
using OrtFloat16 = Microsoft.ML.OnnxRuntime.Float16;
using OrtBFloat16 = Microsoft.ML.OnnxRuntime.BFloat16;
using ShoFloat16 = Shorokoo.Core.Backends.Float16;
using ShoBFloat16 = Shorokoo.Core.Backends.BFloat16;

namespace Shorokoo.OnnxRuntime;

internal sealed class OrtTensorValue : IShorokooTensorValue
{
    internal OrtValue Inner { get; }

    public OrtTensorValue(OrtValue inner) { Inner = inner; }

    public ShorokooOnnxValueType ValueType => (ShorokooOnnxValueType)(int)Inner.OnnxType;

    public ShorokooTensorElementType ElementType =>
        (ShorokooTensorElementType)(int)Inner.GetTensorTypeAndShape().ElementDataType;

    public long[] Shape => Inner.GetTensorTypeAndShape().Shape;

    // ORT names the allocator a value was made by on its memory info, and "Cpu" is the one
    // that names host memory -- every other name ("Cuda", "Hip", ...) is the provider's own.
    // Pinned host memory ("CudaPinned") is readable too, but nothing here ever asks for it,
    // so the narrow test is the safe one: an unrecognized allocator reads as device memory
    // and is copied rather than dereferenced. A value never moves, so this is asked once.
    public bool IsHostAccessible
    {
        get
        {
            if (_probed) return _hostAccessible;
            // Payload before the flag, both volatile: a Nullable<bool> is two fields written
            // non-atomically, so a reader on a weakly ordered target could see HasValue true ahead
            // of the value it stands for. The wrong answer in that direction is a device pointer
            // reported as host-readable and then dereferenced. At worst two threads probe once each
            // and agree -- a value never changes where it lives.
            _hostAccessible = ProbeHostAccessible();
            _probed = true;
            return _hostAccessible;
        }
    }

    private bool ProbeHostAccessible()
    {
        if (!Inner.IsTensor) return false;
        // Disposed, not abandoned: OrtMemoryInfo is a SafeHandle, so leaving one to its finalizer
        // puts an object on the finalization queue for every tensor anyone reads -- the cost
        // OnnxTensorData deliberately refuses to pay by having no finalizer of its own.
        using var info = Inner.GetTensorMemoryInfo();
        return info.Name == CpuAllocatorName;
    }

    private volatile bool _hostAccessible;
    private volatile bool _probed;

    /// <summary>ORT's name for the host allocator, on every execution provider.</summary>
    internal const string CpuAllocatorName = "Cpu";

    /// <summary>Refuses a span over memory the host cannot read. The span accessors hand out a
    /// pointer without checking where it points, so this is the difference between an exception
    /// and a wild read of a device address — and it belongs here rather than only on the tensor
    /// wrapper, because a value reached through <c>ToTensorValue()</c> or copied by
    /// <c>OnnxUtils.CopyTensorValue</c> never passes that wrapper's guard.</summary>
    private void ThrowIfNotHostAccessible()
    {
        if (IsHostAccessible) return;
        // Two different failures share this guard; saying the wrong one sends the reader looking
        // for a resident run that does not exist.
        if (!Inner.IsTensor)
            throw new InvalidOperationException(
                $"This value holds a {Inner.OnnxType}, not a tensor, so it has no element buffer "
                + "to read. Read a sequence through its elements instead.");
        throw new InvalidOperationException(
            "This value's storage is the execution provider's own memory, not host memory, so "
            + "it cannot be read directly. A resident training run leaves its state there "
            + "deliberately; ResidentTrainingRun.StepToCheckpoint is what brings it home.");
    }

    public ReadOnlySpan<T> GetTensorDataAsSpan<T>() where T : unmanaged
    {
        ThrowIfNotHostAccessible();
        if (typeof(T) == typeof(ShoFloat16))
            return MemoryMarshal.Cast<OrtFloat16, T>(Inner.GetTensorDataAsSpan<OrtFloat16>());
        if (typeof(T) == typeof(ShoBFloat16))
            return MemoryMarshal.Cast<OrtBFloat16, T>(Inner.GetTensorDataAsSpan<OrtBFloat16>());
        return Inner.GetTensorDataAsSpan<T>();
    }

    public Span<T> GetTensorMutableDataAsSpan<T>() where T : unmanaged
    {
        ThrowIfNotHostAccessible();
        if (typeof(T) == typeof(ShoFloat16))
            return MemoryMarshal.Cast<OrtFloat16, T>(Inner.GetTensorMutableDataAsSpan<OrtFloat16>());
        if (typeof(T) == typeof(ShoBFloat16))
            return MemoryMarshal.Cast<OrtBFloat16, T>(Inner.GetTensorMutableDataAsSpan<OrtBFloat16>());
        return Inner.GetTensorMutableDataAsSpan<T>();
    }

    public IReadOnlyList<string> GetStringTensorData() => Inner.GetStringTensorAsArray();

    public int GetValueCount() => Inner.GetValueCount();

    public IShorokooTensorValue GetValue(int index) =>
        new OrtTensorValue(Inner.GetValue(index, OrtAllocator.DefaultInstance));

    public ShorokooTensorElementType GetSequenceElementType()
    {
        var info = Inner.GetTypeInfo();
        return (ShorokooTensorElementType)(int)info.SequenceTypeInfo.ElementType.TensorTypeAndShapeInfo.ElementDataType;
    }

    public void Dispose() => Inner.Dispose();
}

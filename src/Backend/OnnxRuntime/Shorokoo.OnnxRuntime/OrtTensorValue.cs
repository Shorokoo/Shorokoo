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

    public ShorokooTensorElementType ElementType
    {
        get
        {
            var elementType = Inner.GetTensorTypeAndShape().ElementDataType;
            GC.KeepAlive(Inner);
            return (ShorokooTensorElementType)(int)elementType;
        }
    }

    public long[] Shape
    {
        get
        {
            var shape = Inner.GetTensorTypeAndShape().Shape;
            GC.KeepAlive(Inner);
            return shape;
        }
    }

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
        // The info ORT hands back does not own what it points at -- it is the tensor's own
        // location, a sub-object of the native value, which OrtReleaseValue frees. So reading
        // info.Name is a native read through Inner's memory, and the keep-alive belongs after it,
        // not after the call that produced the info. Rooted only across the first call, a
        // collection in between frees the value and this reads a dangling pointer -- returning
        // garbage that may compare equal to "Cpu", which is the direction that hands out a span
        // over device memory.
        var host = info.Name == CpuAllocatorName;
        GC.KeepAlive(Inner);
        return host;
    }

    private volatile bool _hostAccessible;
    private volatile bool _probed;

    /// <summary>ORT's name for the host allocator, on every execution provider.</summary>
    internal const string CpuAllocatorName = "Cpu";

    /// <summary>
    /// Whether <paramref name="allocatorName"/> names memory the host owns. Beyond the plain CPU
    /// allocator that is the pinned host arenas a device provider stages copies through: a
    /// device EP serves an output it was asked to leave on the host from its pinned allocator, so
    /// treating that name as device memory reports a graph that partly ran on the host as one that
    /// did not.
    ///
    /// <para>Measured on a CUDA card rather than read off the source. A graph whose tail ORT gave
    /// to the host reports its crossing output as <c>CudaPinned</c> and the host node's own output
    /// as <c>Cpu</c>; a graph that stayed on the card reports <c>Cuda</c> for every output. So the
    /// name appears exactly where the memory is host-readable, and never on an output the provider
    /// kept — the inversion this predicate would suffer from if the premise were backwards.</para>
    /// </summary>
    internal static bool IsHostAllocator(string? allocatorName) =>
        allocatorName is CpuAllocatorName or "CudaPinned" or "HipPinned";

    /// <summary>
    /// Whether this value is a tensor in memory the host cannot read — the execution provider's
    /// own. A value that is not a tensor answers false: it has no buffer of its own to place.
    ///
    /// <para>Deliberately not the negation of <see cref="IsHostAccessible"/>, and the two differ
    /// in both directions. That one answers false for a sequence, which has no element buffer of
    /// its own, where this one asks only about memory; and that one admits the plain CPU allocator
    /// alone, where this one counts the pinned host arenas as host too. The pinned difference is
    /// the load-bearing one: a pinned element is host memory, so ONNX Runtime's host copy reads it
    /// back correctly and there is nothing to refuse — while handing out a span over it is a
    /// wider promise this backend has never measured and does not make.</para>
    ///
    /// <para>Not cached, because it is asked once per value, where a sequence is built.</para>
    /// </summary>
    internal bool IsInDeviceMemory
    {
        get
        {
            if (!Inner.IsTensor) return false;
            using var info = Inner.GetTensorMemoryInfo();
            // After the name read, for the reason ProbeHostAccessible gives: the info points into
            // the native value rather than owning anything.
            var onDevice = !IsHostAllocator(info.Name);
            GC.KeepAlive(Inner);
            return onDevice;
        }
    }

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

    // Each of the four accessors below, and the four reads above them, hands ORT a bare handle off
    // `Inner` and then has no further use for it, so the JIT retires the local at that read --
    // before the native call even starts. OrtValue is a plain class with an ordinary finalizer that
    // calls OrtReleaseValue, so a collection on any thread during the call would free the native
    // value underneath it. Being in scope roots nothing; this does (Shorokoo/Shorokoo#178). The
    // callers happen to keep these values reachable today, which is safety by reachability rather
    // than by construction, and a lifetime changed anywhere above here would quietly take it away.
    // `CoreUtilsCoverageTests.TestEveryNativeCallThroughAnOrtValueKeepsItAliveAndTheGuardStillDetectsEveryEvasion`
    // is what keeps the next such call from being written without one.

    public IReadOnlyList<string> GetStringTensorData()
    {
        var strings = Inner.GetStringTensorAsArray();
        GC.KeepAlive(Inner);
        return strings;
    }

    public int GetValueCount()
    {
        var count = Inner.GetValueCount();
        GC.KeepAlive(Inner);
        return count;
    }

    /// <summary>
    /// The element at <paramref name="index"/>, copied out of this sequence into host memory.
    ///
    /// <para>Host memory unconditionally, because ONNX Runtime offers nothing else: its
    /// <c>GetValue</c> copies the element with a plain host <c>memcpy</c> whatever allocator it is
    /// handed, so an element the execution provider left on a card is read through a device
    /// address by the host and takes the process down with an access violation. That is why
    /// <see cref="OrtBackend.CreateSequence"/> refuses to pack device tensors into a sequence:
    /// the sequence this reads is a host one by construction.</para>
    /// </summary>
    public IShorokooTensorValue GetValue(int index)
    {
        var element = new OrtTensorValue(Inner.GetValue(index, OrtAllocator.DefaultInstance));
        GC.KeepAlive(Inner);
        return element;
    }

    public ShorokooTensorElementType GetSequenceElementType()
    {
        var info = Inner.GetTypeInfo();
        var elementType = info.SequenceTypeInfo.ElementType.TensorTypeAndShapeInfo.ElementDataType;
        GC.KeepAlive(Inner);
        return (ShorokooTensorElementType)(int)elementType;
    }

    public void Dispose() => Inner.Dispose();
}

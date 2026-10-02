namespace Shorokoo.Core.Backends;

// Shorokoo-owned replacement for ORT's OrtValue. The concrete implementation
// lives in the platform DLL (Shorokoo.WinCPU / Shorokoo.WinGPU / Shorokoo.LinuxCPU /
// Shorokoo.LinuxGPU)
// and wraps a real OrtValue; nothing in the main Shorokoo project sees an
// OrtValue directly.
public interface IShorokooTensorValue : IDisposable
{
    ShorokooOnnxValueType ValueType { get; }

    // Only meaningful when ValueType is Tensor.
    ShorokooTensorElementType ElementType { get; }
    long[] Shape { get; }

    // Whether the buffer behind this value is host memory, so the span accessors below
    // may be read. It is false for a value in the execution provider's OWN memory -- a CUDA
    // device allocation, say, which is where a CUDA backend's runs read their inputs and leave
    // their outputs (see IShorokooSession). The span accessors hand out a pointer without
    // checking where it points, so reading one of those spans is not an error but a wild read;
    // callers must consult this first.
    //
    // The default is host memory: a value that does not answer is one the host reads, which is
    // what every value of a backend whose run memory is the host's is.
    bool IsHostAccessible => true;

    ReadOnlySpan<T> GetTensorDataAsSpan<T>() where T : unmanaged;
    Span<T> GetTensorMutableDataAsSpan<T>() where T : unmanaged;

    // Only meaningful when ValueType is Tensor and ElementType is String.
    // Strings are variable-length and reference-typed, so they do not fit the
    // unmanaged<T> span path and need a dedicated accessor.
    IReadOnlyList<string> GetStringTensorData();

    // Only meaningful when ValueType is Sequence.
    int GetValueCount();
    IShorokooTensorValue GetValue(int index);

    // The element type of a sequence's tensor elements. Only meaningful when
    // ValueType is Sequence.
    ShorokooTensorElementType GetSequenceElementType();

    // Where this value stands on a block it shares with other values, each over a range of its
    // own, holding a lease on it (see SharedBlock): a run's output placed in the memory of an input
    // the run consumed. Null for a value that owns its memory whole, which is every value but those.
    internal BlockRange? Range => null;
}

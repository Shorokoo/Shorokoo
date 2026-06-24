namespace Shorokoo.Core.Inference.Abstractions;

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
}

namespace Shorokoo.Core.Inference.Abstractions;

// Implemented once per platform DLL. The platform DLL's identity (WinCPU /
// WinGPU / LinuxCPU / LinuxGPU) determines the EP; there is no EP parameter here.
public interface IShorokooInferenceSessionFactory
{
    IShorokooInferenceSession CreateSession(
        ReadOnlyMemory<byte> modelBytes,
        ShorokooGraphOptimization graphOptimization,
        ShorokooLogSeverity logSeverity);

    IShorokooTensorValue CreateTensor<T>(T[] data, long[] shape) where T : unmanaged;

    IShorokooTensorValue CreateTensorFromRawBytes(
        ShorokooTensorElementType elementType,
        byte[] data,
        long[] shape);

    // String tensors don't fit the raw-bytes path: each element is variable-length
    // UTF-8 and reference-typed, so they get their own constructor.
    IShorokooTensorValue CreateStringTensor(IReadOnlyList<string> data, long[] shape);

    IShorokooTensorValue CreateSequence(IReadOnlyList<IShorokooTensorValue> values);
}
